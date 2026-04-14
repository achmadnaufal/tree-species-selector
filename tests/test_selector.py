"""
Unit tests for src.main.SpeciesSelector.

Run with:
    pytest tests/test_selector.py -v
"""

from __future__ import annotations

import pytest
import pandas as pd

import sys
from pathlib import Path

# Make sure the project root is on sys.path so ``src`` can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import SpeciesSelector, VALID_CLIMATE_ZONES, VALID_SOIL_TYPES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def selector() -> SpeciesSelector:
    """Return a default SpeciesSelector instance."""
    return SpeciesSelector()


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Minimal but realistic species DataFrame used across tests."""
    return pd.DataFrame(
        {
            "species_name": [
                "Teak", "Acacia", "Oak", "Scots Pine", "Neem",
                "Mahogany", "Norway Spruce", "Gmelina",
            ],
            "scientific_name": [
                "Tectona grandis", "Acacia mangium", "Quercus robur",
                "Pinus sylvestris", "Azadirachta indica",
                "Swietenia macrophylla", "Picea abies", "Gmelina arborea",
            ],
            "climate_zone": [
                "tropical", "tropical", "temperate",
                "boreal", "tropical",
                "tropical", "boreal", "tropical",
            ],
            "min_rainfall_mm": [1200, 1000, 500, 300, 400, 1500, 400, 900],
            "max_rainfall_mm": [2500, 3500, 1200, 800, 1000, 3000, 1000, 2800],
            "min_temp_c": [20, 18, 5, -5, 20, 22, 0, 20],
            "max_temp_c": [35, 38, 25, 25, 45, 36, 20, 40],
            "soil_type": [
                "loam", "sandy_loam", "loam",
                "sandy", "sandy_loam",
                "clay_loam", "sandy_loam", "sandy_loam",
            ],
            "growth_rate_m_yr": [1.5, 2.5, 0.6, 0.9, 1.3, 1.2, 0.8, 2.2],
            "carbon_seq_tc_ha_yr": [8.2, 11.4, 4.3, 4.8, 4.5, 9.1, 6.0, 9.8],
            "native": [True, True, True, True, True, False, True, False],
            "drought_tolerant": [True, True, False, True, True, False, False, True],
            "suitable_for_agroforestry": [True, True, False, False, True, True, False, True],
        }
    )


# ---------------------------------------------------------------------------
# 1. Filter by climate zone
# ---------------------------------------------------------------------------

class TestFilterByClimateZone:
    def test_tropical_returns_only_tropical_species(self, selector, sample_df):
        result = selector.filter(sample_df, climate_zone="tropical")
        assert not result.empty
        assert (result["climate_zone"] == "tropical").all()

    def test_boreal_returns_only_boreal_species(self, selector, sample_df):
        result = selector.filter(sample_df, climate_zone="boreal")
        assert len(result) == 2
        assert set(result["species_name"]) == {"Scots Pine", "Norway Spruce"}

    def test_climate_zone_is_case_insensitive(self, selector, sample_df):
        lower = selector.filter(sample_df, climate_zone="tropical")
        upper = selector.filter(sample_df, climate_zone="TROPICAL")
        assert len(lower) == len(upper)

    def test_invalid_climate_zone_raises_value_error(self, selector, sample_df):
        with pytest.raises(ValueError, match="Invalid climate_zone"):
            selector.filter(sample_df, climate_zone="martian")


# ---------------------------------------------------------------------------
# 2. Filter by rainfall range
# ---------------------------------------------------------------------------

class TestFilterByRainfall:
    def test_min_rainfall_excludes_low_max_rainfall_species(self, selector, sample_df):
        # Scots Pine max_rainfall_mm=800; require at least 1000 mm
        result = selector.filter(sample_df, min_rainfall_mm=1000)
        assert "Scots Pine" not in result["species_name"].values

    def test_max_rainfall_excludes_high_min_rainfall_species(self, selector, sample_df):
        # Mahogany min_rainfall_mm=1500; only allow up to 1000 mm
        result = selector.filter(sample_df, max_rainfall_mm=1000)
        assert "Mahogany" not in result["species_name"].values

    def test_combined_rainfall_range_filters_correctly(self, selector, sample_df):
        result = selector.filter(sample_df, min_rainfall_mm=800, max_rainfall_mm=2000)
        # All returned species must have max_rainfall_mm >= 800
        assert (result["max_rainfall_mm"] >= 800).all()
        # All returned species must have min_rainfall_mm <= 2000
        assert (result["min_rainfall_mm"] <= 2000).all()

    def test_negative_rainfall_raises_value_error(self, selector, sample_df):
        with pytest.raises(ValueError, match="min_rainfall_mm must be >= 0"):
            selector.filter(sample_df, min_rainfall_mm=-50)


# ---------------------------------------------------------------------------
# 3. Filter by soil type
# ---------------------------------------------------------------------------

class TestFilterBySoilType:
    def test_loam_returns_only_loam_species(self, selector, sample_df):
        result = selector.filter(sample_df, soil_type="loam")
        assert not result.empty
        assert (result["soil_type"] == "loam").all()

    def test_soil_type_is_case_insensitive(self, selector, sample_df):
        lower = selector.filter(sample_df, soil_type="sandy_loam")
        upper = selector.filter(sample_df, soil_type="SANDY_LOAM")
        assert len(lower) == len(upper)

    def test_invalid_soil_type_raises_value_error(self, selector, sample_df):
        with pytest.raises(ValueError, match="Invalid soil_type"):
            selector.filter(sample_df, soil_type="moon_dust")

    def test_combined_climate_and_soil_filter(self, selector, sample_df):
        result = selector.filter(sample_df, climate_zone="tropical", soil_type="loam")
        assert not result.empty
        assert (result["climate_zone"] == "tropical").all()
        assert (result["soil_type"] == "loam").all()


# ---------------------------------------------------------------------------
# 4. Score / rank calculation
# ---------------------------------------------------------------------------

class TestScoreAndRank:
    def test_score_adds_suitability_score_column(self, selector, sample_df):
        preprocessed = selector.preprocess(sample_df)
        scored = selector.score(preprocessed)
        assert "suitability_score" in scored.columns

    def test_score_values_are_between_0_and_100(self, selector, sample_df):
        preprocessed = selector.preprocess(sample_df)
        scored = selector.score(preprocessed)
        assert (scored["suitability_score"] >= 0).all()
        assert (scored["suitability_score"] <= 100).all()

    def test_rank_orders_by_suitability_descending(self, selector, sample_df):
        ranked = selector.rank(sample_df)
        scores = ranked["suitability_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_rank_adds_rank_column_starting_at_one(self, selector, sample_df):
        ranked = selector.rank(sample_df)
        assert "rank" in ranked.columns
        assert ranked["rank"].iloc[0] == 1

    def test_rank_top_n_limits_results(self, selector, sample_df):
        ranked = selector.rank(sample_df, top_n=3)
        assert len(ranked) == 3

    def test_score_does_not_mutate_input(self, selector, sample_df):
        preprocessed = selector.preprocess(sample_df)
        original_cols = list(preprocessed.columns)
        selector.score(preprocessed)
        assert list(preprocessed.columns) == original_cols

    def test_custom_weights_change_scores(self, selector, sample_df):
        preprocessed = selector.preprocess(sample_df)
        default_scored = selector.score(preprocessed)

        # Selector with all weight on carbon sequestration
        carbon_selector = SpeciesSelector(
            config={
                "score_weights": {
                    "carbon": 1.0,
                    "growth": 0.0,
                    "native": 0.0,
                    "agroforestry": 0.0,
                    "drought": 0.0,
                }
            }
        )
        carbon_scored = carbon_selector.score(preprocessed)

        # Scores must differ for at least one species
        assert not (default_scored["suitability_score"] == carbon_scored["suitability_score"]).all()


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_filter_returns_empty_df_when_no_match(self, selector, sample_df):
        result = selector.filter(sample_df, climate_zone="arid")
        assert result.empty

    def test_score_on_empty_df_raises_value_error(self, selector):
        with pytest.raises(ValueError, match="Cannot score an empty DataFrame"):
            selector.score(pd.DataFrame())

    def test_rank_on_empty_df_raises_value_error(self, selector):
        with pytest.raises(ValueError, match="Cannot rank an empty DataFrame"):
            selector.rank(pd.DataFrame())

    def test_validate_raises_on_empty_df(self, selector):
        with pytest.raises(ValueError, match="empty"):
            selector.validate(pd.DataFrame())

    def test_validate_raises_on_negative_rainfall(self, selector):
        bad_df = pd.DataFrame(
            {
                "species_name": ["Ghost Tree"],
                "min_rainfall_mm": [-100],
                "max_rainfall_mm": [500],
            }
        )
        with pytest.raises(ValueError, match="negative"):
            selector.validate(bad_df)

    def test_validate_raises_when_min_exceeds_max_rainfall(self, selector):
        bad_df = pd.DataFrame(
            {
                "species_name": ["Error Tree"],
                "min_rainfall_mm": [2000],
                "max_rainfall_mm": [500],
            }
        )
        with pytest.raises(ValueError, match="min_rainfall_mm > max_rainfall_mm"):
            selector.validate(bad_df)

    def test_rank_invalid_top_n_raises_value_error(self, selector, sample_df):
        with pytest.raises(ValueError, match="top_n must be a positive integer"):
            selector.rank(sample_df, top_n=0)

    def test_load_data_missing_file_raises_file_not_found(self, selector):
        with pytest.raises(FileNotFoundError):
            selector.load_data("/nonexistent/path/data.csv")

    def test_load_data_unsupported_extension_raises_value_error(self, selector, tmp_path):
        bad_file = tmp_path / "data.json"
        bad_file.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported file format"):
            selector.load_data(str(bad_file))

    def test_filter_does_not_mutate_input_df(self, selector, sample_df):
        original_len = len(sample_df)
        selector.filter(sample_df, climate_zone="tropical")
        assert len(sample_df) == original_len

    def test_preprocess_does_not_mutate_input_df(self, selector, sample_df):
        original_cols = list(sample_df.columns)
        selector.preprocess(sample_df)
        assert list(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# 6. Integration: load sample CSV and run pipeline
# ---------------------------------------------------------------------------

class TestIntegrationWithSampleData:
    def test_load_and_rank_from_csv(self, selector):
        csv_path = Path(__file__).parent.parent / "demo" / "sample_data.csv"
        if not csv_path.exists():
            pytest.skip("demo/sample_data.csv not present")
        df = selector.load_data(str(csv_path))
        ranked = selector.rank(df)
        assert not ranked.empty
        assert "suitability_score" in ranked.columns
        assert "rank" in ranked.columns

    def test_filter_tropical_from_csv(self, selector):
        csv_path = Path(__file__).parent.parent / "demo" / "sample_data.csv"
        if not csv_path.exists():
            pytest.skip("demo/sample_data.csv not present")
        df = selector.load_data(str(csv_path))
        tropical = selector.filter(df, climate_zone="tropical")
        assert not tropical.empty
        assert (tropical["climate_zone"] == "tropical").all()
