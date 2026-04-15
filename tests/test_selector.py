"""
Unit tests for src.main.SpeciesSelector.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=src --cov-report=term-missing
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make sure the project root is on sys.path so ``src`` can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import VALID_CLIMATE_ZONES, VALID_SOIL_TYPES, SpeciesSelector


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
                "Teak",
                "Acacia",
                "Oak",
                "Scots Pine",
                "Neem",
                "Mahogany",
                "Norway Spruce",
                "Gmelina",
            ],
            "scientific_name": [
                "Tectona grandis",
                "Acacia mangium",
                "Quercus robur",
                "Pinus sylvestris",
                "Azadirachta indica",
                "Swietenia macrophylla",
                "Picea abies",
                "Gmelina arborea",
            ],
            "climate_zone": [
                "tropical",
                "tropical",
                "temperate",
                "boreal",
                "tropical",
                "tropical",
                "boreal",
                "tropical",
            ],
            "min_rainfall_mm": [1200, 1000, 500, 300, 400, 1500, 400, 900],
            "max_rainfall_mm": [2500, 3500, 1200, 800, 1000, 3000, 1000, 2800],
            "min_temp_c": [20, 18, 5, -5, 20, 22, 0, 20],
            "max_temp_c": [35, 38, 25, 25, 45, 36, 20, 40],
            "soil_type": [
                "loam",
                "sandy_loam",
                "loam",
                "sandy",
                "sandy_loam",
                "clay_loam",
                "sandy_loam",
                "sandy_loam",
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

    def test_all_valid_climate_zones_are_accepted(self, selector, sample_df):
        """Every zone in VALID_CLIMATE_ZONES should not raise an error."""
        for zone in VALID_CLIMATE_ZONES:
            # Must not raise; result may be empty for zones not in sample_df
            selector.filter(sample_df, climate_zone=zone)

    def test_temperate_zone_returns_correct_species(self, selector, sample_df):
        result = selector.filter(sample_df, climate_zone="temperate")
        assert set(result["species_name"]) == {"Oak"}


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
        assert (result["max_rainfall_mm"] >= 800).all()
        assert (result["min_rainfall_mm"] <= 2000).all()

    def test_negative_rainfall_raises_value_error(self, selector, sample_df):
        with pytest.raises(ValueError, match="min_rainfall_mm must be >= 0"):
            selector.filter(sample_df, min_rainfall_mm=-50)

    def test_negative_max_rainfall_raises_value_error(self, selector, sample_df):
        with pytest.raises(ValueError, match="max_rainfall_mm must be >= 0"):
            selector.filter(sample_df, max_rainfall_mm=-1)

    def test_zero_rainfall_is_valid_boundary(self, selector, sample_df):
        """min_rainfall_mm=0 must not raise."""
        result = selector.filter(sample_df, min_rainfall_mm=0)
        assert not result.empty

    def test_very_high_min_rainfall_returns_empty(self, selector, sample_df):
        result = selector.filter(sample_df, min_rainfall_mm=99999)
        assert result.empty


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

    def test_all_valid_soil_types_are_accepted(self, selector, sample_df):
        """Every type in VALID_SOIL_TYPES should not raise."""
        for soil in VALID_SOIL_TYPES:
            selector.filter(sample_df, soil_type=soil)


# ---------------------------------------------------------------------------
# 4. Filter by boolean flags
# ---------------------------------------------------------------------------


class TestFilterByFlags:
    def test_native_only_returns_native_species(self, selector, sample_df):
        result = selector.filter(sample_df, native_only=True)
        assert not result.empty
        assert (result["native"] == True).all()  # noqa: E712

    def test_drought_tolerant_true_filters_correctly(self, selector, sample_df):
        result = selector.filter(sample_df, drought_tolerant=True)
        assert not result.empty
        assert (result["drought_tolerant"] == True).all()  # noqa: E712

    def test_drought_tolerant_false_filters_correctly(self, selector, sample_df):
        result = selector.filter(sample_df, drought_tolerant=False)
        assert not result.empty
        assert (result["drought_tolerant"] == False).all()  # noqa: E712

    def test_agroforestry_true_filters_correctly(self, selector, sample_df):
        result = selector.filter(sample_df, suitable_for_agroforestry=True)
        assert (result["suitable_for_agroforestry"] == True).all()  # noqa: E712

    def test_combined_flags_narrow_result(self, selector, sample_df):
        result = selector.filter(
            sample_df,
            climate_zone="tropical",
            native_only=True,
            drought_tolerant=True,
        )
        assert not result.empty
        assert (result["climate_zone"] == "tropical").all()
        assert (result["native"] == True).all()  # noqa: E712
        assert (result["drought_tolerant"] == True).all()  # noqa: E712


# ---------------------------------------------------------------------------
# 5. Score / rank calculation
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

    def test_rank_column_is_sequential(self, selector, sample_df):
        ranked = selector.rank(sample_df)
        expected = list(range(1, len(ranked) + 1))
        assert ranked["rank"].tolist() == expected

    def test_rank_top_n_limits_results(self, selector, sample_df):
        ranked = selector.rank(sample_df, top_n=3)
        assert len(ranked) == 3

    def test_rank_top_n_1_returns_single_row(self, selector, sample_df):
        ranked = selector.rank(sample_df, top_n=1)
        assert len(ranked) == 1
        assert ranked["rank"].iloc[0] == 1

    def test_score_does_not_mutate_input(self, selector, sample_df):
        preprocessed = selector.preprocess(sample_df)
        original_cols = list(preprocessed.columns)
        selector.score(preprocessed)
        assert list(preprocessed.columns) == original_cols

    def test_custom_weights_change_scores(self, selector, sample_df):
        preprocessed = selector.preprocess(sample_df)
        default_scored = selector.score(preprocessed)

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
        assert not (default_scored["suitability_score"] == carbon_scored["suitability_score"]).all()

    def test_score_single_row_returns_max_score(self, selector):
        """A single-row DataFrame should receive 100 when all weight columns are True."""
        single = pd.DataFrame(
            {
                "species_name": ["SuperTree"],
                "climate_zone": ["tropical"],
                "carbon_seq_tc_ha_yr": [10.0],
                "growth_rate_m_yr": [2.0],
                "native": [True],
                "drought_tolerant": [True],
                "suitable_for_agroforestry": [True],
            }
        )
        scored = selector.score(single)
        # With a single row, normalise gives 1.0 for numeric columns -> max weighted score
        assert scored["suitability_score"].iloc[0] == pytest.approx(100.0, abs=0.1)


# ---------------------------------------------------------------------------
# 6. Preprocess
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_column_names_lowercased(self, selector):
        df = pd.DataFrame({"SpeciesName": ["Teak"], "ClimateZone": ["Tropical"]})
        result = selector.preprocess(df)
        assert "speciesname" in result.columns
        assert "climatezone" in result.columns

    def test_column_spaces_replaced_with_underscores(self, selector):
        df = pd.DataFrame({"species name": ["Teak"], "climate zone": ["tropical"]})
        result = selector.preprocess(df)
        assert "species_name" in result.columns
        assert "climate_zone" in result.columns

    def test_string_whitespace_stripped(self, selector):
        df = pd.DataFrame({"species_name": ["  Teak  "], "climate_zone": ["tropical"]})
        result = selector.preprocess(df)
        assert result["species_name"].iloc[0] == "Teak"

    def test_climate_zone_normalised_to_lowercase(self, selector):
        df = pd.DataFrame({"climate_zone": ["TROPICAL", "Temperate"]})
        result = selector.preprocess(df)
        assert result["climate_zone"].tolist() == ["tropical", "temperate"]

    def test_boolean_string_true_normalised(self, selector):
        df = pd.DataFrame({"native": ["true"], "species_name": ["Teak"]})
        result = selector.preprocess(df)
        assert result["native"].iloc[0] == True  # noqa: E712

    def test_boolean_string_false_normalised(self, selector):
        df = pd.DataFrame({"drought_tolerant": ["false"], "species_name": ["Pine"]})
        result = selector.preprocess(df)
        assert result["drought_tolerant"].iloc[0] == False  # noqa: E712

    def test_all_nan_rows_dropped(self, selector):
        df = pd.DataFrame(
            {"species_name": ["Teak", None], "climate_zone": ["tropical", None]}
        )
        result = selector.preprocess(df)
        assert len(result) == 1

    def test_preprocess_does_not_mutate_input_df(self, selector, sample_df):
        original_cols = list(sample_df.columns)
        selector.preprocess(sample_df)
        assert list(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# 7. Validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_valid_dataframe_returns_true(self, selector, sample_df):
        assert selector.validate(sample_df) is True

    def test_validate_raises_on_non_dataframe(self, selector):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            selector.validate([1, 2, 3])  # type: ignore[arg-type]

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

    def test_validate_df_without_rainfall_columns_passes(self, selector):
        df = pd.DataFrame({"species_name": ["Oak"], "climate_zone": ["temperate"]})
        assert selector.validate(df) is True


# ---------------------------------------------------------------------------
# 8. Analyze
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_analyze_returns_dict(self, selector, sample_df):
        result = selector.analyze(sample_df)
        assert isinstance(result, dict)

    def test_analyze_total_records_matches_row_count(self, selector, sample_df):
        result = selector.analyze(sample_df)
        assert result["total_records"] == len(sample_df)

    def test_analyze_columns_key_present(self, selector, sample_df):
        result = selector.analyze(sample_df)
        assert "columns" in result

    def test_analyze_missing_pct_key_present(self, selector, sample_df):
        result = selector.analyze(sample_df)
        assert "missing_pct" in result

    def test_analyze_summary_stats_present_for_numeric_data(self, selector, sample_df):
        result = selector.analyze(sample_df)
        assert "summary_stats" in result

    def test_analyze_totals_present_for_numeric_data(self, selector, sample_df):
        result = selector.analyze(sample_df)
        assert "totals" in result

    def test_analyze_means_present_for_numeric_data(self, selector, sample_df):
        result = selector.analyze(sample_df)
        assert "means" in result

    def test_analyze_raises_on_empty_df(self, selector):
        with pytest.raises(ValueError, match="empty"):
            selector.analyze(pd.DataFrame())

    def test_analyze_no_summary_stats_for_non_numeric_df(self, selector):
        string_df = pd.DataFrame({"name": ["Oak", "Teak"], "zone": ["temperate", "tropical"]})
        result = selector.analyze(string_df)
        assert "summary_stats" not in result


# ---------------------------------------------------------------------------
# 9. to_dataframe
# ---------------------------------------------------------------------------


class TestToDataFrame:
    def test_returns_dataframe(self, selector, sample_df):
        analysis = selector.analyze(sample_df)
        result = selector.to_dataframe(analysis)
        assert isinstance(result, pd.DataFrame)

    def test_has_metric_and_value_columns(self, selector, sample_df):
        analysis = selector.analyze(sample_df)
        result = selector.to_dataframe(analysis)
        assert "metric" in result.columns
        assert "value" in result.columns

    def test_flat_dict_produces_one_row_per_key(self, selector):
        flat = {"total_records": 10, "columns": ["a", "b"]}
        result = selector.to_dataframe(flat)
        assert len(result) == 2

    def test_nested_dict_is_flattened_with_dot_separator(self, selector):
        nested = {"totals": {"carbon_seq_tc_ha_yr": 42.0, "growth_rate_m_yr": 5.0}}
        result = selector.to_dataframe(nested)
        assert "totals.carbon_seq_tc_ha_yr" in result["metric"].values
        assert "totals.growth_rate_m_yr" in result["metric"].values

    def test_empty_dict_returns_empty_dataframe(self, selector):
        result = selector.to_dataframe({})
        assert result.empty


# ---------------------------------------------------------------------------
# 10. Edge cases
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

    def test_rank_invalid_top_n_raises_value_error(self, selector, sample_df):
        with pytest.raises(ValueError, match="top_n must be a positive integer"):
            selector.rank(sample_df, top_n=0)

    def test_rank_negative_top_n_raises_value_error(self, selector, sample_df):
        with pytest.raises(ValueError, match="top_n must be a positive integer"):
            selector.rank(sample_df, top_n=-5)

    def test_rank_float_top_n_raises_value_error(self, selector, sample_df):
        with pytest.raises(ValueError, match="top_n must be a positive integer"):
            selector.rank(sample_df, top_n=2.5)  # type: ignore[arg-type]

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

    def test_filter_no_criteria_returns_all_rows(self, selector, sample_df):
        """filter() with no arguments returns a preprocessed copy of the full df."""
        result = selector.filter(sample_df)
        assert len(result) == len(sample_df)

    def test_selector_config_is_immutable_between_instances(self):
        """Config dicts passed in should not bleed between instances."""
        cfg = {"score_weights": {"carbon": 0.5, "growth": 0.5, "native": 0.0, "agroforestry": 0.0, "drought": 0.0}}
        s1 = SpeciesSelector(config=cfg)
        s2 = SpeciesSelector()
        assert s1.config is not s2.config

    def test_score_df_without_scoring_columns_yields_zero_scores(self, selector):
        """If no recognised scoring columns exist, score defaults to 0."""
        df = pd.DataFrame(
            {
                "species_name": ["Unknown A", "Unknown B"],
                "climate_zone": ["tropical", "tropical"],
            }
        )
        scored = selector.score(df)
        assert (scored["suitability_score"] == 0.0).all()


# ---------------------------------------------------------------------------
# 11. Run (full pipeline wrapper)
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_returns_dict(self, selector, tmp_path, sample_df):
        csv_path = tmp_path / "test_data.csv"
        sample_df.to_csv(str(csv_path), index=False)
        result = selector.run(str(csv_path))
        assert isinstance(result, dict)

    def test_run_missing_file_raises_file_not_found(self, selector):
        with pytest.raises(FileNotFoundError):
            selector.run("/no/such/file.csv")

    def test_run_total_records_matches_csv_row_count(self, selector, tmp_path, sample_df):
        csv_path = tmp_path / "run_test.csv"
        sample_df.to_csv(str(csv_path), index=False)
        result = selector.run(str(csv_path))
        assert result["total_records"] == len(sample_df)


# ---------------------------------------------------------------------------
# 12. Integration: load sample CSV and run full pipeline
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

    def test_top5_ranked_species_all_have_scores(self, selector):
        csv_path = Path(__file__).parent.parent / "demo" / "sample_data.csv"
        if not csv_path.exists():
            pytest.skip("demo/sample_data.csv not present")
        df = selector.load_data(str(csv_path))
        top5 = selector.rank(df, top_n=5)
        assert len(top5) == 5
        assert top5["suitability_score"].notna().all()

    def test_full_analysis_pipeline_from_csv(self, selector):
        csv_path = Path(__file__).parent.parent / "demo" / "sample_data.csv"
        if not csv_path.exists():
            pytest.skip("demo/sample_data.csv not present")
        result = selector.run(str(csv_path))
        assert result["total_records"] > 0
        assert "columns" in result
