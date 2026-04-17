"""
Tests for src/site_match_scorer.py.

Covers: happy paths, edge cases (empty input, missing columns, exact-vs-
partial soil match, out-of-envelope decay, boundary values, invalid types),
input validation, immutability, and the recommend_for_site wrapper.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.site_match_scorer import (
    DEFAULT_RAINFALL_TOLERANCE_MM,
    DEFAULT_TEMP_TOLERANCE_C,
    Site,
    recommend_for_site,
    score_site_match,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def species_df() -> pd.DataFrame:
    """Three diverse species spanning tropical / temperate / boreal envelopes."""
    return pd.DataFrame(
        {
            "species_name": ["Teak", "Oak", "Scots Pine"],
            "min_rainfall_mm": [1200, 500, 300],
            "max_rainfall_mm": [2500, 1200, 800],
            "min_temp_c": [20, 5, -5],
            "max_temp_c": [35, 25, 25],
            "soil_type": ["loam", "loam", "sandy"],
        }
    )


@pytest.fixture()
def tropical_site() -> Site:
    """Wet tropical site on loam — perfect match for Teak."""
    return Site(rainfall_mm=1800, temperature_c=27, soil_type="loam", name="Java")


# ---------------------------------------------------------------------------
# Site dataclass
# ---------------------------------------------------------------------------


class TestSite:
    def test_construct_valid_site(self):
        site = Site(rainfall_mm=1500.0, temperature_c=25.0, soil_type="loam")
        assert site.rainfall_mm == 1500.0
        assert site.soil_type == "loam"

    def test_soil_type_normalised_to_lowercase(self):
        site = Site(rainfall_mm=500, temperature_c=10, soil_type="LOAM")
        assert site.soil_type == "loam"

    def test_negative_rainfall_rejected(self):
        with pytest.raises(ValueError, match="rainfall_mm must be >= 0"):
            Site(rainfall_mm=-1.0, temperature_c=20, soil_type="loam")

    def test_invalid_soil_type_rejected(self):
        with pytest.raises(ValueError, match="Invalid soil_type"):
            Site(rainfall_mm=1000, temperature_c=20, soil_type="lava")

    def test_site_is_immutable(self):
        site = Site(rainfall_mm=1000, temperature_c=20, soil_type="loam")
        with pytest.raises(Exception):  # FrozenInstanceError
            site.rainfall_mm = 2000  # type: ignore[misc]


# ---------------------------------------------------------------------------
# score_site_match — happy path
# ---------------------------------------------------------------------------


class TestScoreSiteMatchHappyPath:
    def test_returns_dataframe(self, species_df, tropical_site):
        result = score_site_match(species_df, tropical_site)
        assert isinstance(result, pd.DataFrame)

    def test_adds_expected_columns(self, species_df, tropical_site):
        result = score_site_match(species_df, tropical_site)
        for col in ("rainfall_match", "temperature_match", "soil_match", "site_match_score"):
            assert col in result.columns

    def test_row_count_preserved(self, species_df, tropical_site):
        result = score_site_match(species_df, tropical_site)
        assert len(result) == len(species_df)

    def test_perfect_match_inside_envelope(self, species_df, tropical_site):
        result = score_site_match(species_df, tropical_site)
        teak = result[result["species_name"] == "Teak"].iloc[0]
        assert teak["rainfall_match"] == pytest.approx(1.0)
        assert teak["temperature_match"] == pytest.approx(1.0)
        assert teak["soil_match"] == pytest.approx(1.0)
        assert teak["site_match_score"] == pytest.approx(1.0)

    def test_scores_in_unit_interval(self, species_df, tropical_site):
        result = score_site_match(species_df, tropical_site)
        for col in ("rainfall_match", "temperature_match", "soil_match", "site_match_score"):
            assert (result[col] >= 0.0).all()
            assert (result[col] <= 1.0).all()

    def test_tropical_site_prefers_tropical_species(self, species_df, tropical_site):
        result = score_site_match(species_df, tropical_site)
        teak_score = result[result["species_name"] == "Teak"].iloc[0]["site_match_score"]
        pine_score = result[result["species_name"] == "Scots Pine"].iloc[0]["site_match_score"]
        assert teak_score > pine_score

    def test_input_dataframe_not_mutated(self, species_df, tropical_site):
        snapshot = species_df.copy(deep=True)
        score_site_match(species_df, tropical_site)
        pd.testing.assert_frame_equal(species_df, snapshot)


# ---------------------------------------------------------------------------
# Envelope decay
# ---------------------------------------------------------------------------


class TestEnvelopeDecay:
    def test_score_decays_linearly_outside_envelope(self, species_df):
        # Site rainfall is 250 mm below Teak's min (1200) -> half of 500 mm tolerance
        site = Site(rainfall_mm=950, temperature_c=27, soil_type="loam")
        result = score_site_match(
            species_df, site, rainfall_tolerance_mm=DEFAULT_RAINFALL_TOLERANCE_MM
        )
        teak = result[result["species_name"] == "Teak"].iloc[0]
        assert teak["rainfall_match"] == pytest.approx(0.5, abs=1e-6)

    def test_score_zero_far_outside_envelope(self, species_df):
        # 2000 mm below min -- well past 500 mm tolerance
        site = Site(rainfall_mm=0, temperature_c=27, soil_type="loam")
        result = score_site_match(species_df, site)
        teak = result[result["species_name"] == "Teak"].iloc[0]
        assert teak["rainfall_match"] == 0.0

    def test_temperature_decay_above_max(self, species_df):
        # Teak max_temp_c = 35; site = 37.5 -> 2.5/5.0 deficit -> 0.5 score
        site = Site(rainfall_mm=1800, temperature_c=37.5, soil_type="loam")
        result = score_site_match(
            species_df, site, temp_tolerance_c=DEFAULT_TEMP_TOLERANCE_C
        )
        teak = result[result["species_name"] == "Teak"].iloc[0]
        assert teak["temperature_match"] == pytest.approx(0.5, abs=1e-6)

    def test_boundary_value_at_envelope_edge_scores_one(self, species_df):
        # Exactly at min temperature
        site = Site(rainfall_mm=1800, temperature_c=20, soil_type="loam")
        result = score_site_match(species_df, site)
        teak = result[result["species_name"] == "Teak"].iloc[0]
        assert teak["temperature_match"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Soil matching
# ---------------------------------------------------------------------------


class TestSoilMatching:
    def test_exact_soil_match_scores_one(self, species_df):
        site = Site(rainfall_mm=1800, temperature_c=27, soil_type="loam")
        result = score_site_match(species_df, site)
        teak = result[result["species_name"] == "Teak"].iloc[0]
        assert teak["soil_match"] == 1.0

    def test_compatible_soil_partial_score(self):
        species_df = pd.DataFrame(
            {
                "species_name": ["X"],
                "min_rainfall_mm": [0],
                "max_rainfall_mm": [10000],
                "min_temp_c": [-50],
                "max_temp_c": [50],
                "soil_type": ["clay_loam"],
            }
        )
        site = Site(rainfall_mm=1000, temperature_c=20, soil_type="loam")
        result = score_site_match(species_df, site, soil_partial_score=0.5)
        assert result.iloc[0]["soil_match"] == pytest.approx(0.5)

    def test_unrelated_soil_scores_zero(self):
        species_df = pd.DataFrame(
            {
                "species_name": ["X"],
                "min_rainfall_mm": [0],
                "max_rainfall_mm": [10000],
                "min_temp_c": [-50],
                "max_temp_c": [50],
                "soil_type": ["sandy"],
            }
        )
        site = Site(rainfall_mm=1000, temperature_c=20, soil_type="clay")
        result = score_site_match(species_df, site)
        assert result.iloc[0]["soil_match"] == 0.0


# ---------------------------------------------------------------------------
# Custom weights
# ---------------------------------------------------------------------------


class TestCustomWeights:
    def test_soil_only_weights(self, species_df):
        site = Site(rainfall_mm=0, temperature_c=-50, soil_type="loam")
        result = score_site_match(
            species_df,
            site,
            weights={"rainfall": 0.0, "temperature": 0.0, "soil": 1.0},
        )
        teak = result[result["species_name"] == "Teak"].iloc[0]
        # Soil matches loam exactly -> composite must be 1.0 even though
        # rainfall and temperature are catastrophic.
        assert teak["site_match_score"] == pytest.approx(1.0)

    def test_zero_weights_rejected(self, species_df, tropical_site):
        with pytest.raises(ValueError, match="sum to zero"):
            score_site_match(
                species_df,
                tropical_site,
                weights={"rainfall": 0.0, "temperature": 0.0, "soil": 0.0},
            )

    def test_negative_weights_rejected(self, species_df, tropical_site):
        with pytest.raises(ValueError, match="non-negative"):
            score_site_match(
                species_df,
                tropical_site,
                weights={"rainfall": -1.0, "temperature": 1.0, "soil": 1.0},
            )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_rejects_non_dataframe(self, tropical_site):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            score_site_match([{"species_name": "X"}], tropical_site)  # type: ignore[arg-type]

    def test_rejects_empty_dataframe(self, tropical_site):
        with pytest.raises(ValueError, match="empty"):
            score_site_match(pd.DataFrame(), tropical_site)

    def test_rejects_missing_required_columns(self, tropical_site):
        bad = pd.DataFrame({"species_name": ["X"], "min_rainfall_mm": [100]})
        with pytest.raises(ValueError, match="missing required columns"):
            score_site_match(bad, tropical_site)

    def test_rejects_non_positive_rainfall_tolerance(self, species_df, tropical_site):
        with pytest.raises(ValueError, match="rainfall_tolerance_mm"):
            score_site_match(species_df, tropical_site, rainfall_tolerance_mm=0)

    def test_rejects_non_positive_temp_tolerance(self, species_df, tropical_site):
        with pytest.raises(ValueError, match="temp_tolerance_c"):
            score_site_match(species_df, tropical_site, temp_tolerance_c=-1)

    def test_rejects_soil_partial_score_out_of_range(self, species_df, tropical_site):
        with pytest.raises(ValueError, match="soil_partial_score"):
            score_site_match(species_df, tropical_site, soil_partial_score=1.5)


# ---------------------------------------------------------------------------
# recommend_for_site
# ---------------------------------------------------------------------------


class TestRecommendForSite:
    def test_returns_top_n_rows(self, species_df, tropical_site):
        top = recommend_for_site(species_df, tropical_site, top_n=2)
        assert len(top) == 2

    def test_results_sorted_descending(self, species_df, tropical_site):
        top = recommend_for_site(species_df, tropical_site, top_n=3)
        scores = top["site_match_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_rank_column_starts_at_one(self, species_df, tropical_site):
        top = recommend_for_site(species_df, tropical_site, top_n=3)
        assert list(top["rank"]) == [1, 2, 3]

    def test_min_score_filter_applied(self, species_df):
        # Site that no species fits well; require a high minimum.
        site = Site(rainfall_mm=10000, temperature_c=80, soil_type="peat")
        top = recommend_for_site(species_df, site, top_n=3, min_score=0.9)
        assert len(top) == 0

    def test_invalid_top_n_rejected(self, species_df, tropical_site):
        with pytest.raises(ValueError, match="top_n"):
            recommend_for_site(species_df, tropical_site, top_n=0)

    def test_invalid_min_score_rejected(self, species_df, tropical_site):
        with pytest.raises(ValueError, match="min_score"):
            recommend_for_site(species_df, tropical_site, min_score=1.5)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_input_same_output(species_df, tropical_site):
    r1 = score_site_match(species_df, tropical_site)
    r2 = score_site_match(species_df, tropical_site)
    pd.testing.assert_frame_equal(r1, r2)
