"""
Unit tests for src.portfolio_builder.

Run with:
    pytest tests/test_portfolio_builder.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.portfolio_builder import (
    DEFAULT_TRAIT_COLUMNS,
    PortfolioResult,
    build_portfolio,
    compare_portfolios,
)
from src.site_match_scorer import Site


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def candidate_species() -> pd.DataFrame:
    """Mixed pool of realistic tropical/temperate species."""
    return pd.DataFrame(
        {
            "species_name": [
                "Teak",
                "Acacia",
                "Sengon",
                "Jabon",
                "Mahogany",
                "Neem",
                "Oak",
                "Scots Pine",
            ],
            "min_rainfall_mm": [1200, 1000, 1200, 1500, 1500, 400, 500, 300],
            "max_rainfall_mm": [2500, 3500, 3000, 4000, 3000, 1000, 1200, 800],
            "min_temp_c": [20, 18, 18, 22, 22, 20, 5, -5],
            "max_temp_c": [35, 38, 36, 38, 36, 45, 25, 25],
            "soil_type": [
                "loam",
                "sandy_loam",
                "loam",
                "clay",
                "clay_loam",
                "sandy_loam",
                "loam",
                "sandy",
            ],
            "growth_rate_m_yr": [1.5, 2.5, 3.5, 2.8, 1.2, 1.3, 0.6, 0.9],
            "carbon_seq_tc_ha_yr": [8.2, 11.4, 13.2, 10.7, 9.1, 4.5, 4.3, 4.8],
        }
    )


@pytest.fixture()
def tropical_site() -> Site:
    """Typical tropical planting site in Indonesia."""
    return Site(
        rainfall_mm=1800,
        temperature_c=27,
        soil_type="loam",
        name="West Java block",
    )


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------


class TestBuildPortfolioHappyPath:
    def test_returns_portfolio_result(self, candidate_species, tropical_site):
        result = build_portfolio(candidate_species, tropical_site, portfolio_size=3)
        assert isinstance(result, PortfolioResult)

    def test_portfolio_is_dataframe(self, candidate_species, tropical_site):
        result = build_portfolio(candidate_species, tropical_site, portfolio_size=3)
        assert isinstance(result.portfolio, pd.DataFrame)

    def test_portfolio_size_respected(self, candidate_species, tropical_site):
        result = build_portfolio(candidate_species, tropical_site, portfolio_size=3)
        assert len(result.portfolio) == 3

    def test_portfolio_size_capped_by_eligible_pool(self, candidate_species, tropical_site):
        """Asking for 50 species from a pool of 8 still returns <= 8."""
        result = build_portfolio(candidate_species, tropical_site, portfolio_size=50)
        assert len(result.portfolio) <= len(candidate_species)

    def test_proportions_sum_to_one(self, candidate_species, tropical_site):
        result = build_portfolio(candidate_species, tropical_site, portfolio_size=4)
        assert abs(result.portfolio["proportion"].sum() - 1.0) < 1e-6

    def test_rank_column_starts_at_one_and_is_sequential(self, candidate_species, tropical_site):
        result = build_portfolio(candidate_species, tropical_site, portfolio_size=3)
        assert result.portfolio["rank"].tolist() == [1, 2, 3]

    def test_expected_columns_present(self, candidate_species, tropical_site):
        result = build_portfolio(candidate_species, tropical_site, portfolio_size=3)
        for col in (
            "species_name",
            "site_match_score",
            "diversity_contribution",
            "objective",
            "proportion",
            "rank",
        ):
            assert col in result.portfolio.columns, col

    def test_first_selected_is_highest_site_match(self, candidate_species, tropical_site):
        """The seed species must be the one with the best site_match_score."""
        result = build_portfolio(candidate_species, tropical_site, portfolio_size=3)
        first = result.portfolio.iloc[0]
        # Re-score the eligible pool independently to find the real max
        from src.site_match_scorer import score_site_match

        all_scored = score_site_match(candidate_species, tropical_site)
        top_name = all_scored.sort_values(
            "site_match_score", ascending=False
        ).iloc[0]["species_name"]
        assert first["species_name"] == top_name

    def test_first_selected_has_zero_diversity_contribution(
        self, candidate_species, tropical_site
    ):
        result = build_portfolio(candidate_species, tropical_site, portfolio_size=3)
        assert result.portfolio.iloc[0]["diversity_contribution"] == 0.0

    def test_mean_site_match_is_non_negative(self, candidate_species, tropical_site):
        result = build_portfolio(candidate_species, tropical_site, portfolio_size=3)
        assert 0.0 <= result.mean_site_match <= 1.0

    def test_summary_is_non_empty_string(self, candidate_species, tropical_site):
        result = build_portfolio(candidate_species, tropical_site, portfolio_size=3)
        assert isinstance(result.summary, str) and result.summary


# ---------------------------------------------------------------------------
# 2. alpha behaviour
# ---------------------------------------------------------------------------


class TestAlphaBehaviour:
    def test_alpha_1_ranks_purely_by_site_fit(self, candidate_species, tropical_site):
        """alpha=1.0 should order species monotonically by site_match_score."""
        result = build_portfolio(
            candidate_species, tropical_site, portfolio_size=4, alpha=1.0
        )
        scores = result.portfolio["site_match_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_alpha_0_prefers_diverse_choice_over_best_fit(
        self, candidate_species, tropical_site
    ):
        """With alpha=0 the 2nd pick should not necessarily be the 2nd-best fit."""
        fit_only = build_portfolio(
            candidate_species, tropical_site, portfolio_size=3, alpha=1.0
        )
        div_only = build_portfolio(
            candidate_species, tropical_site, portfolio_size=3, alpha=0.0
        )
        # Seeds must match (both start from highest site fit)
        assert fit_only.portfolio.iloc[0]["species_name"] == div_only.portfolio.iloc[0]["species_name"]
        # But the second choice should differ for a rich-enough trait space
        # (if it doesn't differ, the test still passes as long as the sets can differ)
        fit_set = set(fit_only.portfolio["species_name"])
        div_set = set(div_only.portfolio["species_name"])
        assert fit_set != div_set or fit_only.portfolio.iloc[1]["species_name"] != div_only.portfolio.iloc[1]["species_name"]

    def test_alpha_boundary_values_accepted(self, candidate_species, tropical_site):
        for a in (0.0, 0.5, 1.0):
            result = build_portfolio(
                candidate_species, tropical_site, portfolio_size=2, alpha=a
            )
            assert len(result.portfolio) == 2
            assert result.alpha == pytest.approx(a)


# ---------------------------------------------------------------------------
# 3. Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_non_dataframe_raises_type_error(self, tropical_site):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            build_portfolio([1, 2, 3], tropical_site)  # type: ignore[arg-type]

    def test_empty_dataframe_raises_value_error(self, tropical_site):
        with pytest.raises(ValueError, match="empty"):
            build_portfolio(pd.DataFrame(), tropical_site)

    @pytest.mark.parametrize("bad_size", [0, -1, -100])
    def test_non_positive_portfolio_size_raises(self, candidate_species, tropical_site, bad_size):
        with pytest.raises(ValueError, match="portfolio_size"):
            build_portfolio(candidate_species, tropical_site, portfolio_size=bad_size)

    def test_float_portfolio_size_raises(self, candidate_species, tropical_site):
        with pytest.raises(ValueError, match="portfolio_size"):
            build_portfolio(candidate_species, tropical_site, portfolio_size=2.5)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_alpha", [-0.1, 1.5, 42.0])
    def test_out_of_range_alpha_raises(self, candidate_species, tropical_site, bad_alpha):
        with pytest.raises(ValueError, match="alpha must be in"):
            build_portfolio(candidate_species, tropical_site, alpha=bad_alpha)

    @pytest.mark.parametrize("bad_min", [-0.01, 1.01, 2.0])
    def test_out_of_range_min_site_score_raises(
        self, candidate_species, tropical_site, bad_min
    ):
        with pytest.raises(ValueError, match="min_site_score"):
            build_portfolio(candidate_species, tropical_site, min_site_score=bad_min)

    def test_missing_required_columns_propagates(self, tropical_site):
        bad = pd.DataFrame({"species_name": ["Foo"], "min_rainfall_mm": [1000]})
        with pytest.raises(ValueError, match="missing required columns"):
            build_portfolio(bad, tropical_site)


# ---------------------------------------------------------------------------
# 4. Edge cases and no-match scenarios
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_candidates_meet_min_site_score_returns_empty(
        self, candidate_species, tropical_site
    ):
        # Site in Sahara-style arid conditions where no candidate fits
        arid_site = Site(
            rainfall_mm=50,
            temperature_c=45,
            soil_type="sandy",
            name="arid outlier",
        )
        result = build_portfolio(
            candidate_species, arid_site, min_site_score=0.95
        )
        assert result.portfolio.empty
        assert result.mean_site_match == 0.0
        assert "No candidates" in result.summary

    def test_single_candidate_pool_returns_single_species(self, tropical_site):
        single = pd.DataFrame(
            {
                "species_name": ["Teak"],
                "min_rainfall_mm": [1200],
                "max_rainfall_mm": [2500],
                "min_temp_c": [20],
                "max_temp_c": [35],
                "soil_type": ["loam"],
                "growth_rate_m_yr": [1.5],
                "carbon_seq_tc_ha_yr": [8.2],
            }
        )
        result = build_portfolio(single, tropical_site, portfolio_size=5)
        assert len(result.portfolio) == 1
        assert result.portfolio.iloc[0]["proportion"] == pytest.approx(1.0)

    def test_build_does_not_mutate_input_dataframe(self, candidate_species, tropical_site):
        original_cols = list(candidate_species.columns)
        original_len = len(candidate_species)
        build_portfolio(candidate_species, tropical_site, portfolio_size=3)
        assert list(candidate_species.columns) == original_cols
        assert len(candidate_species) == original_len

    def test_missing_trait_columns_still_works(self, tropical_site):
        """If no usable trait columns are supplied, diversity is 0 for everyone.

        With alpha=0 the objective is purely diversity (all zeros) so
        proportions must fall back to an equal split, and no exception is
        raised from the zero-width trait matrix.
        """
        minimal = pd.DataFrame(
            {
                "species_name": ["Teak", "Acacia", "Sengon"],
                "min_rainfall_mm": [1200, 1000, 1200],
                "max_rainfall_mm": [2500, 3500, 3000],
                "min_temp_c": [20, 18, 18],
                "max_temp_c": [35, 38, 36],
                "soil_type": ["loam", "sandy_loam", "loam"],
            }
        )
        # Explicitly pass an empty trait-column tuple to force zero-width traits
        result = build_portfolio(
            minimal,
            tropical_site,
            portfolio_size=3,
            alpha=0.0,
            trait_columns=(),
        )
        assert len(result.portfolio) == 3
        for prop in result.portfolio["proportion"].tolist():
            assert prop == pytest.approx(1.0 / 3, abs=1e-3)

    def test_result_is_frozen(self, candidate_species, tropical_site):
        result = build_portfolio(candidate_species, tropical_site, portfolio_size=2)
        with pytest.raises((AttributeError, Exception)):
            result.alpha = 999  # type: ignore[misc]

    def test_deterministic_output(self, candidate_species, tropical_site):
        """Two runs with identical inputs produce identical portfolios."""
        r1 = build_portfolio(candidate_species, tropical_site, portfolio_size=4)
        r2 = build_portfolio(candidate_species, tropical_site, portfolio_size=4)
        pd.testing.assert_frame_equal(r1.portfolio, r2.portfolio)


# ---------------------------------------------------------------------------
# 5. compare_portfolios
# ---------------------------------------------------------------------------


class TestComparePortfolios:
    def test_returns_one_row_per_alpha(self, candidate_species, tropical_site):
        alphas = [0.0, 0.3, 0.7, 1.0]
        out = compare_portfolios(candidate_species, tropical_site, alphas=alphas)
        assert len(out) == len(alphas)
        assert out["alpha"].tolist() == alphas

    def test_expected_columns_present(self, candidate_species, tropical_site):
        out = compare_portfolios(
            candidate_species, tropical_site, alphas=[0.2, 0.8]
        )
        for col in (
            "alpha",
            "n_species",
            "weighted_site_fit",
            "mean_diversity_contribution",
            "species_names",
        ):
            assert col in out.columns

    def test_empty_alphas_raises(self, candidate_species, tropical_site):
        with pytest.raises(ValueError, match="at least one"):
            compare_portfolios(candidate_species, tropical_site, alphas=[])

    def test_invalid_alpha_raises(self, candidate_species, tropical_site):
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            compare_portfolios(
                candidate_species, tropical_site, alphas=[0.5, 2.0]
            )

    def test_alpha_1_never_produces_lower_site_fit_than_alpha_0(
        self, candidate_species, tropical_site
    ):
        """Pure site-fit (alpha=1) should weakly dominate pure diversity on fit."""
        out = compare_portfolios(
            candidate_species,
            tropical_site,
            alphas=[0.0, 1.0],
            portfolio_size=3,
        )
        fit_row = out[out["alpha"] == 1.0].iloc[0]
        div_row = out[out["alpha"] == 0.0].iloc[0]
        assert fit_row["weighted_site_fit"] >= div_row["weighted_site_fit"] - 1e-9


# ---------------------------------------------------------------------------
# 6. Default trait columns sanity check
# ---------------------------------------------------------------------------


def test_default_trait_columns_includes_carbon_and_growth():
    assert "carbon_seq_tc_ha_yr" in DEFAULT_TRAIT_COLUMNS
    assert "growth_rate_m_yr" in DEFAULT_TRAIT_COLUMNS
