"""
Tests for src/species_diversity_scorer.py.

Covers: happy paths, edge cases (empty input, single species, zero proportions,
out-of-range values, missing trait columns), determinism, and parametrized cases.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.species_diversity_scorer import (
    DEFAULT_TRAIT_COLUMNS,
    DiversityResult,
    compute_diversity,
    score_plan_diversity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def three_species_plan() -> pd.DataFrame:
    """Balanced three-species plan with full trait data."""
    return pd.DataFrame(
        {
            "species_name": ["Teak", "Sengon", "Albizzia"],
            "proportion": [0.5, 0.3, 0.2],
            "growth_rate_m_yr": [1.5, 3.5, 2.0],
            "carbon_seq_tc_ha_yr": [8.2, 13.2, 7.5],
            "min_rainfall_mm": [1200, 800, 900],
            "max_rainfall_mm": [2500, 3000, 2800],
            "min_temp_c": [20, 18, 18],
            "max_temp_c": [35, 38, 36],
        }
    )


@pytest.fixture()
def uniform_plan() -> pd.DataFrame:
    """Four species with equal proportions — maximum evenness."""
    return pd.DataFrame(
        {
            "species_name": ["A", "B", "C", "D"],
            "proportion": [0.25, 0.25, 0.25, 0.25],
            "growth_rate_m_yr": [1.0, 2.0, 3.0, 4.0],
            "carbon_seq_tc_ha_yr": [5.0, 8.0, 11.0, 14.0],
        }
    )


@pytest.fixture()
def single_species_plan() -> pd.DataFrame:
    """Single species — edge case for evenness and functional diversity."""
    return pd.DataFrame(
        {
            "species_name": ["Teak"],
            "proportion": [1.0],
            "growth_rate_m_yr": [1.5],
            "carbon_seq_tc_ha_yr": [8.2],
        }
    )


# ---------------------------------------------------------------------------
# Happy path — compute_diversity
# ---------------------------------------------------------------------------


class TestComputeDiversityHappyPath:
    def test_returns_diversity_result_instance(self, three_species_plan):
        result = compute_diversity(three_species_plan)
        assert isinstance(result, DiversityResult)

    def test_species_count_correct(self, three_species_plan):
        result = compute_diversity(three_species_plan)
        assert result.species_count == 3

    def test_shannon_is_positive(self, three_species_plan):
        result = compute_diversity(three_species_plan)
        assert result.shannon_index > 0.0

    def test_simpson_in_valid_range(self, three_species_plan):
        result = compute_diversity(three_species_plan)
        assert 0.0 <= result.simpson_index < 1.0

    def test_evenness_in_valid_range(self, three_species_plan):
        result = compute_diversity(three_species_plan)
        assert result.evenness is not None
        assert 0.0 <= result.evenness <= 1.0

    def test_functional_diversity_is_positive(self, three_species_plan):
        result = compute_diversity(three_species_plan)
        assert result.functional_diversity is not None
        assert result.functional_diversity > 0.0

    def test_summary_contains_shannon(self, three_species_plan):
        result = compute_diversity(three_species_plan)
        assert "Shannon" in result.summary

    def test_uniform_plan_evenness_near_one(self, uniform_plan):
        result = compute_diversity(uniform_plan)
        assert result.evenness is not None
        assert result.evenness == pytest.approx(1.0, abs=1e-6)

    def test_uniform_plan_shannon_equals_log_s(self, uniform_plan):
        result = compute_diversity(uniform_plan)
        expected = math.log(4)
        assert result.shannon_index == pytest.approx(expected, abs=1e-6)

    def test_unnormalised_proportions_accepted(self):
        """Proportions given as raw areas (e.g., hectares) are normalised internally."""
        plan = pd.DataFrame(
            {
                "species_name": ["A", "B"],
                "proportion": [30.0, 70.0],  # raw hectares
                "growth_rate_m_yr": [1.0, 2.0],
            }
        )
        result = compute_diversity(plan)
        # Equivalent to proportions 0.3 / 0.7
        reference = compute_diversity(
            pd.DataFrame(
                {
                    "species_name": ["A", "B"],
                    "proportion": [0.3, 0.7],
                    "growth_rate_m_yr": [1.0, 2.0],
                }
            )
        )
        assert result.shannon_index == pytest.approx(reference.shannon_index, abs=1e-9)

    def test_input_dataframe_not_mutated(self, three_species_plan):
        original_proportions = three_species_plan["proportion"].copy()
        compute_diversity(three_species_plan)
        pd.testing.assert_series_equal(three_species_plan["proportion"], original_proportions)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_species_shannon_is_zero(self, single_species_plan):
        result = compute_diversity(single_species_plan)
        assert result.shannon_index == pytest.approx(0.0, abs=1e-9)

    def test_single_species_evenness_is_none(self, single_species_plan):
        result = compute_diversity(single_species_plan)
        assert result.evenness is None

    def test_single_species_functional_diversity_is_none(self, single_species_plan):
        result = compute_diversity(single_species_plan)
        assert result.functional_diversity is None

    def test_plan_without_trait_columns_returns_none_functional_diversity(self):
        plan = pd.DataFrame(
            {
                "species_name": ["A", "B", "C"],
                "proportion": [0.5, 0.3, 0.2],
            }
        )
        result = compute_diversity(plan, trait_columns=DEFAULT_TRAIT_COLUMNS)
        assert result.functional_diversity is None

    def test_zero_proportion_rows_excluded_from_species_count(self):
        plan = pd.DataFrame(
            {
                "species_name": ["A", "B", "C"],
                "proportion": [0.5, 0.5, 0.0],  # C is absent
                "growth_rate_m_yr": [1.0, 2.0, 3.0],
            }
        )
        result = compute_diversity(plan)
        assert result.species_count == 2

    def test_custom_proportion_column_name(self):
        plan = pd.DataFrame(
            {
                "species_name": ["A", "B"],
                "area_ha": [60.0, 40.0],
                "growth_rate_m_yr": [1.0, 2.0],
            }
        )
        result = compute_diversity(plan, proportion_col="area_ha")
        assert result.species_count == 2
        assert result.shannon_index > 0.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_output(self, three_species_plan):
        r1 = compute_diversity(three_species_plan)
        r2 = compute_diversity(three_species_plan)
        assert r1.shannon_index == r2.shannon_index
        assert r1.simpson_index == r2.simpson_index
        assert r1.functional_diversity == r2.functional_diversity


# ---------------------------------------------------------------------------
# Input validation errors
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_raises_type_error_for_non_dataframe(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            compute_diversity([{"species_name": "A", "proportion": 1.0}])  # type: ignore[arg-type]

    def test_raises_value_error_for_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            compute_diversity(pd.DataFrame())

    def test_raises_value_error_for_missing_proportion_column(self):
        plan = pd.DataFrame({"species_name": ["A"], "area": [1.0]})
        with pytest.raises(ValueError, match="proportion"):
            compute_diversity(plan)

    def test_raises_value_error_for_negative_proportion(self):
        plan = pd.DataFrame({"species_name": ["A", "B"], "proportion": [-0.1, 1.1]})
        with pytest.raises(ValueError, match="negative"):
            compute_diversity(plan)

    def test_raises_value_error_when_all_proportions_zero(self):
        plan = pd.DataFrame({"species_name": ["A", "B"], "proportion": [0.0, 0.0]})
        with pytest.raises(ValueError, match="zero"):
            compute_diversity(plan)


# ---------------------------------------------------------------------------
# Parametrized cases — known Shannon values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "proportions, expected_shannon",
    [
        ([1.0], 0.0),  # single species
        ([0.5, 0.5], math.log(2)),  # two equal species
        ([1 / 3, 1 / 3, 1 / 3], math.log(3)),  # three equal species
    ],
)
def test_shannon_known_values(proportions, expected_shannon):
    plan = pd.DataFrame(
        {
            "species_name": [f"sp_{i}" for i in range(len(proportions))],
            "proportion": proportions,
        }
    )
    result = compute_diversity(plan)
    assert result.shannon_index == pytest.approx(expected_shannon, abs=1e-9)


# ---------------------------------------------------------------------------
# score_plan_diversity
# ---------------------------------------------------------------------------


class TestScorePlanDiversity:
    def test_returns_dataframe(self, three_species_plan):
        df = score_plan_diversity(three_species_plan)
        assert isinstance(df, pd.DataFrame)

    def test_has_exactly_one_row(self, three_species_plan):
        df = score_plan_diversity(three_species_plan)
        assert len(df) == 1

    def test_expected_columns_present(self, three_species_plan):
        df = score_plan_diversity(three_species_plan)
        expected = {
            "species_count",
            "shannon_index",
            "simpson_index",
            "evenness",
            "functional_diversity",
        }
        assert expected.issubset(set(df.columns))

    def test_values_match_compute_diversity(self, three_species_plan):
        result = compute_diversity(three_species_plan)
        df = score_plan_diversity(three_species_plan)
        assert df["shannon_index"].iloc[0] == pytest.approx(result.shannon_index, abs=1e-5)
        assert df["simpson_index"].iloc[0] == pytest.approx(result.simpson_index, abs=1e-5)

    def test_single_species_evenness_is_none_in_df(self, single_species_plan):
        df = score_plan_diversity(single_species_plan)
        assert df["evenness"].iloc[0] is None
