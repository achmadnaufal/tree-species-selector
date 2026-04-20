"""
Unit tests for src.climate_envelope.

Run with:
    pytest tests/test_climate_envelope.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.climate_envelope import (
    BIOCLIM_VARIABLES,
    DEFAULT_BIO_WEIGHTS,
    BioclimRange,
    ClimateEnvelopeMatcher,
    EnvelopeMatchResult,
    SpeciesEnvelope,
    categorise_score,
    widen_envelope,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def teak_envelope() -> SpeciesEnvelope:
    """Envelope for Teak (Tectona grandis), tropical SE Asia."""
    return SpeciesEnvelope(
        species_name="Teak",
        scientific_name="Tectona grandis",
        ranges={
            "bio1": BioclimRange("bio1", 22.0, 28.0),
            "bio5": BioclimRange("bio5", 30.0, 40.0),
            "bio6": BioclimRange("bio6", 13.0, 24.0),
            "bio12": BioclimRange("bio12", 1200.0, 2500.0),
            "bio14": BioclimRange("bio14", 20.0, 120.0),
            "bio17": BioclimRange("bio17", 80.0, 350.0),
        },
    )


@pytest.fixture()
def tropical_site() -> dict:
    """A tropical SE Asia site well-matched to Teak."""
    return {
        "bio1": 26.5,
        "bio5": 33.0,
        "bio6": 21.0,
        "bio12": 2100.0,
        "bio14": 45.0,
        "bio17": 180.0,
    }


@pytest.fixture()
def matcher() -> ClimateEnvelopeMatcher:
    return ClimateEnvelopeMatcher()


@pytest.fixture()
def envelope_csv_path() -> Path:
    return Path(__file__).parent.parent / "sample_data" / "climate_envelope_samples.csv"


# ---------------------------------------------------------------------------
# 1. BioclimRange
# ---------------------------------------------------------------------------


class TestBioclimRange:
    def test_valid_range_constructs(self) -> None:
        rng = BioclimRange("bio1", 20.0, 30.0)
        assert rng.low == 20.0
        assert rng.high == 30.0

    def test_invalid_variable_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown BIO variable"):
            BioclimRange("bio99", 0.0, 10.0)

    def test_low_greater_than_high_raises(self) -> None:
        with pytest.raises(ValueError, match="must be <="):
            BioclimRange("bio1", 30.0, 20.0)

    def test_width_property(self) -> None:
        rng = BioclimRange("bio12", 1000.0, 2500.0)
        assert rng.width == pytest.approx(1500.0)

    def test_score_inside_range_is_one(self) -> None:
        rng = BioclimRange("bio1", 20.0, 30.0)
        assert rng.score(25.0) == 1.0
        assert rng.score(20.0) == 1.0
        assert rng.score(30.0) == 1.0

    def test_score_just_outside_is_between_zero_and_one(self) -> None:
        rng = BioclimRange("bio1", 20.0, 30.0)
        s = rng.score(30.5)
        assert 0.0 < s < 1.0

    def test_score_far_outside_is_zero(self) -> None:
        rng = BioclimRange("bio1", 20.0, 30.0)
        # 10 C below low, well beyond 25% buffer of width=10 -> 0
        assert rng.score(5.0) == 0.0

    def test_score_is_symmetric_around_range(self) -> None:
        rng = BioclimRange("bio1", 20.0, 30.0)
        assert rng.score(19.0) == pytest.approx(rng.score(31.0))

    def test_immutability_of_range(self) -> None:
        rng = BioclimRange("bio1", 20.0, 30.0)
        with pytest.raises((AttributeError, Exception)):
            rng.low = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. SpeciesEnvelope
# ---------------------------------------------------------------------------


class TestSpeciesEnvelope:
    def test_valid_envelope_constructs(self, teak_envelope: SpeciesEnvelope) -> None:
        assert teak_envelope.species_name == "Teak"
        assert "bio1" in teak_envelope.ranges

    def test_empty_species_name_raises(self) -> None:
        with pytest.raises(ValueError, match="species_name"):
            SpeciesEnvelope(species_name="  ", scientific_name="Foo bar")

    def test_empty_scientific_name_raises(self) -> None:
        with pytest.raises(ValueError, match="scientific_name"):
            SpeciesEnvelope(species_name="Foo", scientific_name="")

    def test_mismatched_range_key_raises(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            SpeciesEnvelope(
                species_name="Foo",
                scientific_name="Foo bar",
                ranges={"bio1": BioclimRange("bio5", 0.0, 10.0)},
            )


# ---------------------------------------------------------------------------
# 3. ClimateEnvelopeMatcher - weight validation
# ---------------------------------------------------------------------------


class TestMatcherWeights:
    def test_default_weights_sum_to_one(self) -> None:
        assert abs(sum(DEFAULT_BIO_WEIGHTS.values()) - 1.0) < 1e-9

    def test_custom_weights_accepted(self) -> None:
        w = {k: 1.0 / 6 for k in BIOCLIM_VARIABLES}
        m = ClimateEnvelopeMatcher(weights=w)
        assert m is not None

    def test_weights_missing_key_raises(self) -> None:
        w = {k: 0.2 for k in list(BIOCLIM_VARIABLES)[:-1]}
        with pytest.raises(ValueError, match="must contain exactly"):
            ClimateEnvelopeMatcher(weights=w)

    def test_weights_not_summing_to_one_raises(self) -> None:
        w = {k: 0.5 for k in BIOCLIM_VARIABLES}
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ClimateEnvelopeMatcher(weights=w)

    def test_negative_weight_raises(self) -> None:
        w = {k: 1.0 / 6 for k in BIOCLIM_VARIABLES}
        w["bio1"] = -0.5
        w["bio12"] = 0.5 + (1.0 / 6)
        with pytest.raises(ValueError, match="must be >= 0"):
            ClimateEnvelopeMatcher(weights=w)


# ---------------------------------------------------------------------------
# 4. match_site
# ---------------------------------------------------------------------------


class TestMatchSite:
    def test_perfect_match_scores_100(
        self, matcher: ClimateEnvelopeMatcher, teak_envelope: SpeciesEnvelope, tropical_site: dict
    ) -> None:
        result = matcher.match_site(tropical_site, teak_envelope)
        assert result.match_score == pytest.approx(100.0)
        assert result.category == "excellent"

    def test_partial_match_below_100(
        self, matcher: ClimateEnvelopeMatcher, teak_envelope: SpeciesEnvelope
    ) -> None:
        # Too cold: bio1 below range
        site = {"bio1": 10.0, "bio5": 33.0, "bio6": 21.0, "bio12": 2100, "bio14": 45, "bio17": 180}
        result = matcher.match_site(site, teak_envelope)
        assert result.match_score < 100.0

    def test_missing_bio_variable_raises(
        self, matcher: ClimateEnvelopeMatcher, teak_envelope: SpeciesEnvelope
    ) -> None:
        incomplete = {"bio1": 25.0}
        with pytest.raises(ValueError, match="missing required"):
            matcher.match_site(incomplete, teak_envelope)

    def test_non_dict_site_raises_type_error(
        self, matcher: ClimateEnvelopeMatcher, teak_envelope: SpeciesEnvelope
    ) -> None:
        with pytest.raises(TypeError, match="dict"):
            matcher.match_site([1, 2, 3], teak_envelope)  # type: ignore[arg-type]

    def test_negative_precipitation_raises(
        self, matcher: ClimateEnvelopeMatcher, teak_envelope: SpeciesEnvelope, tropical_site: dict
    ) -> None:
        bad_site = {**tropical_site, "bio12": -50}
        with pytest.raises(ValueError, match="precipitation"):
            matcher.match_site(bad_site, teak_envelope)

    def test_non_numeric_value_raises(
        self, matcher: ClimateEnvelopeMatcher, teak_envelope: SpeciesEnvelope, tropical_site: dict
    ) -> None:
        bad_site = {**tropical_site, "bio1": "hot"}
        with pytest.raises(ValueError, match="must be numeric"):
            matcher.match_site(bad_site, teak_envelope)

    def test_result_category_labels(
        self, matcher: ClimateEnvelopeMatcher, teak_envelope: SpeciesEnvelope, tropical_site: dict
    ) -> None:
        result = matcher.match_site(tropical_site, teak_envelope)
        assert result.category in {"excellent", "good", "marginal", "unsuitable"}

    def test_per_variable_scores_populated(
        self, matcher: ClimateEnvelopeMatcher, teak_envelope: SpeciesEnvelope, tropical_site: dict
    ) -> None:
        result = matcher.match_site(tropical_site, teak_envelope)
        assert set(result.per_variable_scores.keys()) == set(BIOCLIM_VARIABLES)

    def test_empty_envelope_ranges_scores_zero(
        self, matcher: ClimateEnvelopeMatcher, tropical_site: dict
    ) -> None:
        empty_env = SpeciesEnvelope(
            species_name="Empty",
            scientific_name="Empty sp",
            ranges={},
        )
        result = matcher.match_site(tropical_site, empty_env)
        assert result.match_score == 0.0
        assert result.category == "unsuitable"


# ---------------------------------------------------------------------------
# 5. rank_species
# ---------------------------------------------------------------------------


class TestRankSpecies:
    def test_ranked_in_descending_order(
        self, matcher: ClimateEnvelopeMatcher, tropical_site: dict, envelope_csv_path: Path
    ) -> None:
        envelopes = ClimateEnvelopeMatcher.load_envelopes(str(envelope_csv_path))
        ranked = matcher.rank_species(tropical_site, envelopes)
        scores = [r.match_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_limits_results(
        self, matcher: ClimateEnvelopeMatcher, tropical_site: dict, envelope_csv_path: Path
    ) -> None:
        envelopes = ClimateEnvelopeMatcher.load_envelopes(str(envelope_csv_path))
        ranked = matcher.rank_species(tropical_site, envelopes, top_n=3)
        assert len(ranked) == 3

    def test_empty_envelopes_raises(
        self, matcher: ClimateEnvelopeMatcher, tropical_site: dict
    ) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            matcher.rank_species(tropical_site, [])

    def test_invalid_top_n_raises(
        self, matcher: ClimateEnvelopeMatcher, tropical_site: dict, teak_envelope: SpeciesEnvelope
    ) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            matcher.rank_species(tropical_site, [teak_envelope], top_n=0)

    def test_teak_ranks_highly_for_tropical_se_asia(
        self, matcher: ClimateEnvelopeMatcher, tropical_site: dict, envelope_csv_path: Path
    ) -> None:
        envelopes = ClimateEnvelopeMatcher.load_envelopes(str(envelope_csv_path))
        ranked = matcher.rank_species(tropical_site, envelopes, top_n=5)
        species_top5 = {r.species_name for r in ranked}
        assert "Teak" in species_top5


# ---------------------------------------------------------------------------
# 6. Loading & DataFrame conversion
# ---------------------------------------------------------------------------


class TestLoadAndDataFrame:
    def test_load_envelopes_from_csv(self, envelope_csv_path: Path) -> None:
        envelopes = ClimateEnvelopeMatcher.load_envelopes(str(envelope_csv_path))
        assert len(envelopes) >= 15
        assert all(isinstance(e, SpeciesEnvelope) for e in envelopes)

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            ClimateEnvelopeMatcher.load_envelopes("/nonexistent/file.csv")

    def test_to_dataframe_columns(
        self,
        matcher: ClimateEnvelopeMatcher,
        tropical_site: dict,
        envelope_csv_path: Path,
    ) -> None:
        envelopes = ClimateEnvelopeMatcher.load_envelopes(str(envelope_csv_path))
        ranked = matcher.rank_species(tropical_site, envelopes, top_n=3)
        df = matcher.to_dataframe(ranked)
        assert list(df.columns[:5]) == [
            "rank",
            "species_name",
            "scientific_name",
            "match_score",
            "category",
        ]
        assert len(df) == 3


# ---------------------------------------------------------------------------
# 7. Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_categorise_excellent(self) -> None:
        assert categorise_score(95.0) == "excellent"

    def test_categorise_good(self) -> None:
        assert categorise_score(70.0) == "good"

    def test_categorise_marginal(self) -> None:
        assert categorise_score(50.0) == "marginal"

    def test_categorise_unsuitable(self) -> None:
        assert categorise_score(10.0) == "unsuitable"

    def test_categorise_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            categorise_score(150.0)

    def test_widen_envelope_increases_range_width(
        self, teak_envelope: SpeciesEnvelope
    ) -> None:
        widened = widen_envelope(teak_envelope, 0.1)
        for key, rng in teak_envelope.ranges.items():
            assert widened.ranges[key].width > rng.width

    def test_widen_envelope_does_not_mutate_input(
        self, teak_envelope: SpeciesEnvelope
    ) -> None:
        original_bio1_low = teak_envelope.ranges["bio1"].low
        widen_envelope(teak_envelope, 0.25)
        assert teak_envelope.ranges["bio1"].low == original_bio1_low

    def test_widen_envelope_negative_factor_raises(
        self, teak_envelope: SpeciesEnvelope
    ) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            widen_envelope(teak_envelope, -0.1)

    def test_widen_envelope_zero_factor_returns_equivalent(
        self, teak_envelope: SpeciesEnvelope
    ) -> None:
        widened = widen_envelope(teak_envelope, 0.0)
        for key, rng in teak_envelope.ranges.items():
            assert widened.ranges[key].low == rng.low
            assert widened.ranges[key].high == rng.high
