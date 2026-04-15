"""
Unit tests for src.data_generator module.

Run with:
    pytest tests/test_data_generator.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generator import (
    _LEGACY_COLUMNS,
    _SPECIES_POOL,
    generate_domain_sample,
    generate_sample,
)


# ---------------------------------------------------------------------------
# generate_sample
# ---------------------------------------------------------------------------


class TestGenerateSample:
    def test_returns_dataframe(self):
        result = generate_sample(10)
        assert isinstance(result, pd.DataFrame)

    def test_row_count_matches_n(self):
        for n in (1, 10, 50, 300):
            assert len(generate_sample(n)) == n

    def test_columns_match_legacy_schema(self):
        df = generate_sample(10)
        for col in _LEGACY_COLUMNS:
            assert col in df.columns

    def test_reproducible_with_same_seed(self):
        df1 = generate_sample(50, seed=7)
        df2 = generate_sample(50, seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        df1 = generate_sample(50, seed=1)
        df2 = generate_sample(50, seed=2)
        # At least one cell should differ
        assert not df1.equals(df2)

    def test_invalid_n_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="positive integer"):
            generate_sample(0)

    def test_invalid_n_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="positive integer"):
            generate_sample(-5)

    def test_invalid_n_float_raises_value_error(self):
        with pytest.raises(ValueError, match="positive integer"):
            generate_sample(10.5)  # type: ignore[arg-type]

    def test_no_null_values_in_output(self):
        df = generate_sample(50)
        assert not df.isnull().any().any()


# ---------------------------------------------------------------------------
# generate_domain_sample
# ---------------------------------------------------------------------------


class TestGenerateDomainSample:
    def test_returns_dataframe(self):
        result = generate_domain_sample()
        assert isinstance(result, pd.DataFrame)

    def test_row_count_matches_species_pool(self):
        result = generate_domain_sample()
        assert len(result) == len(_SPECIES_POOL)

    def test_required_columns_present(self):
        required = {
            "species_name",
            "scientific_name",
            "climate_zone",
            "min_rainfall_mm",
            "max_rainfall_mm",
            "soil_type",
            "growth_rate_m_yr",
            "carbon_seq_tc_ha_yr",
            "native",
            "drought_tolerant",
            "suitable_for_agroforestry",
        }
        result = generate_domain_sample()
        assert required.issubset(set(result.columns))

    def test_climate_zones_are_valid_strings(self):
        result = generate_domain_sample()
        valid_zones = {"tropical", "subtropical", "temperate", "boreal", "arid"}
        assert set(result["climate_zone"]).issubset(valid_zones)

    def test_rainfall_min_lte_max(self):
        result = generate_domain_sample()
        assert (result["min_rainfall_mm"] <= result["max_rainfall_mm"]).all()

    def test_does_not_mutate_species_pool(self):
        original_first = dict(_SPECIES_POOL[0])
        generate_domain_sample()
        assert dict(_SPECIES_POOL[0]) == original_first

    def test_custom_pool_overrides_default(self):
        custom_pool = (
            {
                "species_name": "Test Tree",
                "scientific_name": "Testus treeicus",
                "climate_zone": "tropical",
                "min_rainfall_mm": 500,
                "max_rainfall_mm": 1500,
                "min_temp_c": 20,
                "max_temp_c": 35,
                "soil_type": "loam",
                "growth_rate_m_yr": 1.0,
                "carbon_seq_tc_ha_yr": 5.0,
                "native": True,
                "drought_tolerant": False,
                "suitable_for_agroforestry": True,
            },
        )
        result = generate_domain_sample(species_pool=custom_pool)
        assert len(result) == 1
        assert result["species_name"].iloc[0] == "Test Tree"

    def test_growth_rate_is_positive(self):
        result = generate_domain_sample()
        assert (result["growth_rate_m_yr"] > 0).all()

    def test_carbon_seq_is_positive(self):
        result = generate_domain_sample()
        assert (result["carbon_seq_tc_ha_yr"] > 0).all()
